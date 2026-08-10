"""
Migrate anonymous cases to authenticated user.

This script assigns all cases with user_id=None to a specific user.
Useful for migrating data after fixing the authentication issue.
"""
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import get_settings
from app.models import Case, User

settings = get_settings()

# Create database session
engine = create_engine(settings.DATABASE_URL.replace('db', 'localhost'))
SessionLocal = sessionmaker(bind=engine)


def migrate_anonymous_cases(target_user_email: str = None):
    """
    Migrate all anonymous cases to a target user.
    
    Args:
        target_user_email: Email of the user to assign cases to.
                          If None, will use the first user in the system.
    """
    db = SessionLocal()
    
    try:
        # Get target user
        if target_user_email:
            target_user = db.query(User).filter(User.email == target_user_email).first()
            if not target_user:
                print(f"Error: User with email '{target_user_email}' not found.")
                return
        else:
            # Get first user in system
            target_user = db.query(User).first()
            if not target_user:
                print("Error: No users found in the system.")
                return
        
        print(f"Target user: {target_user.name} ({target_user.email})")
        
        # Find all anonymous cases
        anonymous_cases = db.query(Case).filter(Case.user_id == None).all()
        
        if not anonymous_cases:
            print("No anonymous cases found. Nothing to migrate.")
            return
        
        print(f"Found {len(anonymous_cases)} anonymous cases.")
        
        # Confirm migration
        response = input(f"\nMigrate {len(anonymous_cases)} anonymous cases to user '{target_user.email}'? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return
        
        # Migrate cases
        migrated_count = 0
        for case in anonymous_cases:
            case.user_id = target_user.id
            migrated_count += 1
        
        # Commit changes
        db.commit()
        
        print(f"\n✅ Successfully migrated {migrated_count} cases to user '{target_user.email}'")
        print(f"   User ID: {target_user.id}")
        
        # Show summary
        total_cases = db.query(Case).filter(Case.user_id == target_user.id).count()
        print(f"   Total cases for this user: {total_cases}")
        
    except Exception as e:
        db.rollback()
        print(f"Error: {str(e)}")
    finally:
        db.close()


def show_stats():
    """Show current database statistics."""
    db = SessionLocal()
    
    try:
        total_users = db.query(User).count()
        total_cases = db.query(Case).count()
        anonymous_cases = db.query(Case).filter(Case.user_id == None).count()
        
        print("\n=== Database Statistics ===")
        print(f"Total users: {total_users}")
        print(f"Total cases: {total_cases}")
        print(f"Anonymous cases: {anonymous_cases}")
        print(f"Assigned cases: {total_cases - anonymous_cases}")
        
        if total_users > 0:
            print("\n=== Users ===")
            users = db.query(User).all()
            for user in users:
                user_cases = db.query(Case).filter(Case.user_id == user.id).count()
                print(f"  - {user.name} ({user.email}): {user_cases} cases")
        
    finally:
        db.close()


if __name__ == "__main__":
    print("=== Anonymous Cases Migration Tool ===\n")
    
    # Show current stats
    show_stats()
    
    # Check if there are anonymous cases
    db = SessionLocal()
    anonymous_count = db.query(Case).filter(Case.user_id == None).count()
    db.close()
    
    if anonymous_count == 0:
        print("\n✅ No anonymous cases to migrate!")
        sys.exit(0)
    
    print("\n" + "="*50)
    
    # Ask for target user email
    target_email = input("\nEnter the email of the user to assign cases to (or press Enter for first user): ").strip()
    
    if target_email:
        migrate_anonymous_cases(target_email)
    else:
        migrate_anonymous_cases()
    
    # Show updated stats
    print("\n" + "="*50)
    show_stats()
