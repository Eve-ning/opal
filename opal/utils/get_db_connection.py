from sqlalchemy import create_engine, Connection


# Using mysql-connector-python
# mysql+mysqlconnector://<user>:<password>@<host>[:<port>]/<dbname>


def get_db_connection(
        db_name: str,
        user_name: str = "root",
        password: str = "p@ssw0rd1",
        container_name: str = "osu-mysql",
        port: int = 3307
) -> Connection:
    """ Returns the docker container osu-mysql connection """
    engine = create_engine(
        f'mysql+mysqlconnector://{user_name}:{password}@{container_name}:{port}/{db_name}'
    )
    return engine.connect()
