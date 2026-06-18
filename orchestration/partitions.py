from dagster import DynamicPartitionsDefinition

adunbox_account_partition = DynamicPartitionsDefinition(name="adunbox_account_ids")
vibelets_account_partition = DynamicPartitionsDefinition(name="vibelets_account_ids")
